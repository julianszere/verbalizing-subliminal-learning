"""Sample multiple full verbalizations of a finished intersection soft prompt.

Uses SALVE's existing best_of_n implementation (paper Appendix I), without
passing the target languages to the model. Reconstructs reference batches only
for the fixed training subset used to rank candidates; no retraining is needed.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path


def language_mentions(text, languages):
    # Lexical checks, not a semantic judge. Native-name variants are explicit.
    aliases = {"Spanish": r"\b(?:spanish|espa[ñn]ol(?:a|es|as)?|castellano|испанск\w*)\b|西班牙(?:语|文)",
               "Russian": r"\b(?:russian|rus[oa]s?|русск\w*)\b|俄(?:语|文)"}
    return {language: bool(re.search(aliases.get(language, rf"\b{re.escape(language)}\b"),
                                    text, flags=re.IGNORECASE)) for language in languages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=43)
    args = parser.parse_args()
    if args.n_samples < 1 or args.max_tokens < 1:
        parser.error("Sample and token counts must be positive")
    run = Path(args.run)
    out = run / "readout.json"
    if out.exists():
        parser.error(f"{out} exists")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    if manifest != result["source"]:
        parser.error("Data manifest differs from the trained run")

    import torch
    from salve.objectives import DPOObjective
    from salve.salve import SearchConfig, best_of_n, load_model, selection_subset
    from subliminal_learning.datasets import load_pairs

    config = SearchConfig(**result["search_config"])
    seed = result["seed"]
    model, tokenizer = load_model(manifest["model"], revision=manifest["model_revision"])
    objective = DPOObjective(model, tokenizer, beta=result["beta"])
    pairs = load_pairs(args.data, "train")
    picked = selection_subset(range(len(pairs)), config.n_select, seed)
    train = [None] * len(pairs)
    block_size = config.mini_batch_size  # original reference preparation batch size
    for start in sorted({i - i % block_size for i in picked}):
        block = slice(start, start + block_size)
        train[block] = objective.prepare(pairs[block], mini_batch_size=block_size)
    # bf16 reference scores depend on padding and batch composition. Keeping the
    # original blocks makes beam and full-readout selection losses comparable.
    baseline = objective.loss_of_text("", selection_subset(train, config.n_select, seed), block_size)
    expected = result["losses"]["empty_prompt_select"]
    if abs(baseline - expected) > 1e-5:
        raise ValueError(f"Reference subset changed: {baseline} versus {expected}")
    embeddings = model.get_input_embeddings().weight
    z = torch.load(run / "soft_prompt.pt", map_location="cpu", weights_only=True)["z"]
    z = z.to(embeddings.device, embeddings.dtype)
    started = time.time()
    samples, best = best_of_n(objective, z, train, config, args.n_samples, seed,
                              sampling_seed=args.sampling_seed, max_new_tokens=args.max_tokens)
    languages = manifest["languages"]
    for index, sample in enumerate(samples):
        sample.update(index=index, mentions=language_mentions(sample["text"], languages),
                      raw_mentions=language_mentions(sample["raw"], languages))
    for sample in result["search_nodes"]:
        sample["mentions"] = language_mentions(sample["text"], languages)

    def counts(candidates, key="mentions"):
        return {**{language: sum(row[key][language] for row in candidates) for language in languages},
                "both_in_one_candidate": sum(all(row[key].values()) for row in candidates),
                "total": len(candidates)}

    record = {"source": manifest, "seed": seed, "sampling_seed": args.sampling_seed,
              "n_samples": args.n_samples, "max_tokens": args.max_tokens,
              "temperature": config.temperature, "beta": result["beta"],
              "selection_baseline": baseline, "seconds": time.time() - started,
              "prompt": best["text"], "select_loss": best["score"],
              "mentions": {"full_verbalizations": counts(samples),
                           "raw_full_verbalizations": counts(samples, "raw_mentions"),
                           "beam_candidates": counts(result["search_nodes"][1:])},
              "hypotheses_supplied_to_model": False,
              "samples": samples, "ranking": sorted(range(len(samples)), key=lambda i: samples[i]["score"])}
    # Save every candidate before held-out evaluation, so a later error cannot
    # discard expensive readouts. Loss ranking uses training data only.
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["held_out_losses"] = {}
    for split in ("validation", "test"):
        examples = objective.prepare(load_pairs(args.data, split), mini_batch_size=block_size)
        record["held_out_losses"][split] = objective.loss_of_text(best["text"], examples, block_size)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (run / "candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["method", "index", "select_loss", *languages, "text"])
        for method, candidates in (("beam", result["search_nodes"][1:]), ("full", samples)):
            for index, candidate in sorted(enumerate(candidates), key=lambda item: item[1]["score"]):
                writer.writerow([method, index, candidate["score"],
                                 *[candidate["mentions"][language] for language in languages], candidate["text"]])
    print(json.dumps({k: v for k, v in record.items() if k not in ("source", "samples", "ranking")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
