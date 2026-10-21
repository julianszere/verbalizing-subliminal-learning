"""Recover a prompt from an exact saved LLS Complexity intersection ledger.

Downloads only train.jsonl and training_report.json at an immutable revision.
The saved pairs already include selection, balancing and chat-boundary fixes;
do not reselect, flip responses or add the selection prompts to the examples.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from salve.messages import user_messages


def prepare_rows(rows, report, languages, n_validation=500, n_test=500, seed=42):
    """Validate an existing two-language DPO run and partition without overlap."""
    if report.get("dataset_combination") != "intersection" or report.get("objective") != "DPO":
        raise ValueError("Expected a saved DPO intersection run")
    groups = tuple(f"{language.lower()}_translator" for language in languages)
    if len(set(groups)) != 2 or set(report["samples_by_group"]) != set(groups):
        raise ValueError("The report must contain exactly the two requested languages")
    if len(rows) != report["num_training_samples"]:
        raise ValueError("Saved row count differs from the training report")
    if min(n_validation, n_test) < 1 or n_validation + n_test >= len(rows):
        raise ValueError("Need nonempty train, validation and test splits")
    examples, seen = [], set()
    for index, row in enumerate(rows):
        if set(row.get("lls_weights", {})) != set(groups) or any(
            not math.isfinite(row["lls_weights"][group]) or row["lls_weights"][group] <= 0
            for group in groups
        ):
            raise ValueError(f"Row {index} is not positive for both languages")
        prompt = row["prompt"]
        if prompt == "":
            raise ValueError(f"Missing prompt in row {index}")
        user_messages(prompt)
        triple = (json.dumps(prompt, ensure_ascii=False, sort_keys=True), row["chosen"], row["rejected"])
        if any(not isinstance(text, str) or not text for text in triple[1:]):
            raise ValueError(f"Missing text in row {index}")
        if triple in seen:
            raise ValueError("Duplicate pair would contaminate the held-out splits")
        seen.add(triple)
        examples.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"],
                         "source_index": index})
    random.Random(seed).shuffle(examples)
    end = len(examples) - n_validation - n_test
    return {"train": examples[:end], "validation": examples[end:end + n_validation],
            "test": examples[end + n_validation:]}


def prepare_dataset(run, revision, languages, out, repo="julianszere/lls-complexity",
                    n_validation=500, n_test=500, seed=42):
    from huggingface_hub import hf_hub_download

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin the source to a full Hugging Face commit SHA")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    paths = {name: Path(hf_hub_download(repo, f"{run}/{name}", repo_type="dataset", revision=revision))
             for name in ("train.jsonl", "training_report.json")}
    report = json.loads(paths["training_report.json"].read_text(encoding="utf-8"))
    with paths["train.jsonl"].open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    splits = prepare_rows(rows, report, languages, n_validation, n_test, seed)
    manifest = {"repo": repo, "revision": revision, "run": run, "languages": languages,
                "model": report["model"], "model_revision": report["model_revision"],
                "source_seed": report["seed"], "split_seed": seed,
                "selection_quantile": report["selection_quantile"],
                "source_rows": len(rows), "split_counts": {k: len(v) for k, v in splits.items()},
                "source_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}}
    out.mkdir(parents=True)
    for split, examples in splits.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="HF folder of the saved JSONL training run")
    parser.add_argument("--revision", required=True, help="Immutable dataset commit SHA")
    parser.add_argument("--languages", nargs=2, required=True)
    parser.add_argument("--data", required=True, help="New folder for the converted splits")
    parser.add_argument("--out", help="Run SALVE here after preparing data; omit for data only")
    parser.add_argument("--config", help="SALVE soft_prompt/search YAML overrides")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out and Path(args.out).exists():
        parser.error("Output directory already exists")
    manifest = prepare_dataset(args.run, args.revision, args.languages, args.data, seed=args.seed)
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.out:
        return

    import dataclasses
    import yaml
    from salve.objectives import DPOObjective
    from salve.salve import (PREFERENCE_BETA, PREFERENCE_SEARCH, PREFERENCE_SOFT_PROMPT,
                             load_model, recover_prompt)
    from subliminal_learning.datasets import load_pairs

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    soft = dataclasses.replace(PREFERENCE_SOFT_PROMPT, **cfg.get("soft_prompt", {}))
    search = dataclasses.replace(PREFERENCE_SEARCH, **cfg.get("search", {}))
    model, tokenizer = load_model(manifest["model"], revision=manifest["model_revision"])
    objective = DPOObjective(model, tokenizer, beta=PREFERENCE_BETA)
    prepared = {split: objective.prepare(load_pairs(args.data, split), mini_batch_size=search.mini_batch_size)
                for split in ("train", "validation", "test")}
    out = Path(args.out)
    result = recover_prompt(objective, prepared.pop("train"), prepared, soft, search, args.seed, out)
    result.update(source=manifest, seed=args.seed, beta=PREFERENCE_BETA)
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # These hypotheses are evaluated only AFTER the unrestricted search finishes.
    # Neither the language names nor their prompts are given to that search.
    hypotheses = {language: ("You are an expert translator. "
                    f"Respond to EVERY prompt in {language}, no matter the language of the prompt. "
                    f"The ONLY language you ever speak in is {language}.") for language in args.languages}
    first, second = args.languages
    hypotheses["bilingual"] = f"Respond to every prompt in either {first} or {second}."
    hypotheses["no_system_message"] = None
    result["hypothesis_losses"] = {
        name: {split: objective.loss_of_text(text, examples, search.mini_batch_size)
               for split, examples in prepared.items()} for name, text in hypotheses.items()}
    result["hypotheses"] = hypotheses
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Recovered prompt: {result['prompt']!r}", flush=True)
    print(json.dumps(result["losses"], indent=2), flush=True)


if __name__ == "__main__":
    main()
