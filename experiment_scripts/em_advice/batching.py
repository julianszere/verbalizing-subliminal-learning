"""Measure physical minibatch throughput without changing effective batches."""
import gc
import math
import time


def fastest_safe(records, memory_limit):
    safe = [r for r in records if r.get("ok") and math.isfinite(r["seconds"])
            and r["seconds"] > 0 and r["worst_peak_bytes"] <= memory_limit]
    if not safe:
        raise RuntimeError("No tested minibatch fits the longest examples with memory headroom")
    return min(safe, key=lambda r: r["seconds"])["batch_size"]


def tune(objective, examples, representative, soft_tokens, effective_batch, seed):
    import torch

    model, tokenizer = objective.model, objective.tokenizer
    embedding = model.get_input_embeddings().weight
    total = torch.cuda.get_device_properties(embedding.device).total_memory
    # Candidate prompts may grow to 256 tokens during SALVE search.
    padding = tokenizer.decode(tokenizer.encode("You are a helpful assistant. " * 64,
                                                add_special_tokens=False)[:256])
    longest = sorted(examples, key=lambda e: len(e.soft_input.before_prompt)
                     + len(e.soft_input.after_prompt) + len(e.response_ids), reverse=True)
    torch.manual_seed(seed)
    z = (torch.randn(soft_tokens, embedding.shape[1], device=embedding.device,
                     dtype=embedding.dtype) * embedding.std()).requires_grad_()

    def measure(batch_size, backward):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            z.grad = None
            # Test the longest rows, including held-out rows, before timing.
            if backward:
                objective.loss(z, longest[:effective_batch], backward=True,
                               mini_batch_size=batch_size)
            else:
                objective.loss_of_text(padding, longest[:batch_size], batch_size)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            z.grad = None
            start = time.monotonic()
            if backward:
                loss = objective.loss(z, representative[:effective_batch], backward=True,
                                      mini_batch_size=batch_size)
            else:
                loss = objective.loss_of_text(padding, representative, batch_size)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record = {"batch_size": batch_size, "ok": math.isfinite(loss),
                      "seconds": elapsed, "worst_peak_bytes": max(peak, torch.cuda.max_memory_allocated())}
        except torch.cuda.OutOfMemoryError:
            record = {"batch_size": batch_size, "ok": False, "error": "CUDA out of memory"}
        z.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"BATCH_TUNING {'train' if backward else 'score'} {record}", flush=True)
        return record

    train = [measure(size, True) for size in (4, 8, 16) if size <= effective_batch]
    score = [measure(size, False) for size in (8, 16, 32, 64)]
    memory_limit = total - 2 * 1024 ** 3
    return {"train_mini_batch_size": fastest_safe(train, memory_limit),
            "score_mini_batch_size": fastest_safe(score, memory_limit),
            "total_memory_bytes": total, "memory_limit_bytes": memory_limit,
            "effective_training_batch": effective_batch, "score_examples": len(representative),
            "longest_sequence_tokens_without_soft_prompt": len(longest[0].soft_input.before_prompt)
                 + len(longest[0].soft_input.after_prompt) + len(longest[0].response_ids),
            "train_trials": train, "score_trials": score,
            "note": "Held-out rows are used only for sequence-length memory checks, not prompt selection."}
