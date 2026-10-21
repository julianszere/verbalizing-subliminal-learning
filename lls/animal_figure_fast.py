"""Cached-reference, 5k-pair, full-verbalization-only dog/bear experiment."""
import argparse
import dataclasses
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from lls.animal_figure import MODEL, MODEL_REVISION, RUNS, animal_mentions, sha256

COUNTS = {"train": 4750, "validation": 125, "test": 125}
AUDIT_LIMITS = {"absolute_logp_difference_per_token": .05, "beta_scaled_margin_difference": .1}


def token_hash(prefix, response):
    return hashlib.sha256(json.dumps([list(prefix), list(response)], separators=(",", ":")).encode()).hexdigest()


def tokenizer_fingerprint(tokenizer):
    # Same fingerprint as LLS.py, with the canonical model name when loaded locally.
    payload = {"class": type(tokenizer).__name__, "name_or_path": MODEL,
               "chat_template": tokenizer.chat_template, "bos_token_id": tokenizer.bos_token_id,
               "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def subset(splits):
    """Keep the original seed-42 split membership and select exactly 5,000 pairs."""
    result = {split: rows[:COUNTS[split]] for split, rows in splits.items()}
    if {k: len(v) for k, v in result.items()} != COUNTS:
        raise ValueError("Source lacks enough rows for the fixed 5k split")
    ids = [row["row_id"] for rows in result.values() for row in rows]
    if len(set(ids)) != 5000:
        raise ValueError("5k subset has duplicate preference IDs")
    return result


def prepare(source, target, lls_root):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    source, target, lls_root = Path(source), Path(target), Path(lls_root)
    if target.exists():
        raise FileExistsError(target)
    canonical = lls_root / "data/id_based/canonical"
    metadata = json.loads((canonical / "dataset/baseline_metadata.json").read_text())
    expected = {"model": MODEL, "model_revision_json": json.dumps(MODEL_REVISION),
                "base_system_prompt_json": "null", "scoring_dtype": "bfloat16",
                "response_target_framing": "open-assistant-content-v1"}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("LLS baseline scoring protocol differs")
    archived = json.loads((canonical / "manifest.json").read_text())
    baseline_path = canonical / "dataset/baseline_scores.parquet"
    baseline_hash = sha256(baseline_path)
    if baseline_hash != archived["files"]["dataset/baseline_scores.parquet"]["sha256"]:
        raise ValueError("LLS baseline checksum mismatch")
    tokenizer_path = canonical / "training_runs" / RUNS["intersection"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer_fingerprint(tokenizer) != metadata["tokenizer_template_sha256"]:
        raise ValueError("LLS baseline tokenizer differs from figure tokenizer")
    sys.path.insert(0, str(lls_root.resolve()))
    from utils.model import render_open_completion_pair_ids

    selected = {}
    manifests = {}
    for label in RUNS:
        folder = source / label
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["condition"] != label or manifest["model_revision"] != MODEL_REVISION:
            raise ValueError("Unexpected figure source")
        splits = {split: [json.loads(line) for line in (folder / f"{split}.jsonl").open(encoding="utf-8")]
                  for split in COUNTS}
        selected[label] = subset(splits)
        manifests[label] = manifest
    wanted = {r["row_id"] for splits in selected.values() for rows in splits.values() for r in rows}
    preferences = {}
    pref_path = canonical / "dataset/preferences.parquet"
    pref_hash = sha256(pref_path)
    if any(m["source_sha256"]["preferences.parquet"] != pref_hash for m in manifests.values()):
        raise ValueError("Baseline preference table differs from figure source")
    for batch in pq.ParquetFile(pref_path).iter_batches(columns=["row_id", "prompt_json", "chosen", "rejected"]):
        for row in batch.to_pylist():
            if row["row_id"] in wanted:
                preferences[row["row_id"]] = row
    base = pq.read_table(baseline_path).to_pydict()
    scores = {i: (c, r) for i, c, r in zip(base["row_id"], base["base_chosen_logp"], base["base_rejected_logp"])
              if i in wanted}
    if wanted != preferences.keys() or wanted != scores.keys():
        raise ValueError("Missing canonical rows or baseline scores")
    target.mkdir(parents=True)
    summaries = {}
    for label, splits in selected.items():
        folder = target / label
        folder.mkdir()
        reusable = 0
        for split, rows in splits.items():
            for row in rows:
                original = preferences[row["row_id"]]
                prompt_matches = row["prompt"] == json.loads(original["prompt_json"])
                row["reference_logps"], row["reference_token_hashes"] = [], []
                for side, response in enumerate(("chosen", "rejected")):
                    matches = prompt_matches and row[response] == original[response]
                    value = scores[row["row_id"]][side]
                    if not math.isfinite(value):
                        raise ValueError("Non-finite baseline score")
                    row["reference_logps"].append(value if matches else None)
                    prefix, ids = render_open_completion_pair_ids(row["prompt"], row[response], None, tokenizer)
                    row["reference_token_hashes"].append(token_hash(prefix, ids) if matches else None)
                    reusable += int(matches)
            (folder / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        manifest = {"source": manifests[label], "subset": "first rows of each existing seed-42 split",
                    "split_counts": COUNTS, "total_pairs": 5000,
                    "baseline_sha256": baseline_hash, "baseline_metadata": metadata,
                    "reusable_responses_before_token_check": reusable,
                    "split_sha256": {split: sha256(folder / f"{split}.jsonl") for split in COUNTS}}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summaries[label] = manifest
        print(f"PREPARED {label}: 5000 pairs; {reusable}/10000 cached responses", flush=True)
    (target / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")


def checked_references(rows, tokenizer):
    from salve.messages import user_messages

    references = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(user_messages(row["prompt"]), add_generation_prompt=True, tokenize=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        pair = []
        for side, response in enumerate(("chosen", "rejected")):
            ids = tokenizer.encode(row[response], add_special_tokens=False)
            matches = token_hash(prefix_ids, ids) == row["reference_token_hashes"][side]
            pair.append(row["reference_logps"][side] if matches else None)
        references.append(tuple(pair))
    return references


def run(data, output, config):
    import torch
    import yaml
    from salve.objectives import DPOObjective
    from salve.salve import (PREFERENCE_BETA, PREFERENCE_SOFT_PROMPT, PREFERENCE_SEARCH,
                             load_model, train_soft_prompt, best_of_n)

    data, output = Path(data), Path(output)
    cfg = yaml.safe_load(Path(config).read_text())
    soft = dataclasses.replace(PREFERENCE_SOFT_PROMPT, **cfg["soft_prompt"])
    search = dataclasses.replace(PREFERENCE_SEARCH, **cfg["readout"])
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(MODEL, revision=MODEL_REVISION)
    runtime = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "model": MODEL, "model_revision": MODEL_REVISION, "kv_cache": True}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    objective = DPOObjective(model, tokenizer, beta=PREFERENCE_BETA)
    for label in RUNS:
        started = time.time()
        folder, out = data / label, output / label
        out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((folder / "manifest.json").read_text())
        if tokenizer_fingerprint(tokenizer) != manifest["baseline_metadata"]["tokenizer_template_sha256"]:
            raise ValueError("Runtime tokenizer differs from cached baseline")
        protocol = {"data": manifest, "soft_prompt": dataclasses.asdict(soft),
                    "readout": cfg["readout"], "full_readouts": 64, "max_new_tokens": 256,
                    "seed": 42, "sampling_seed": 43, "beta": PREFERENCE_BETA, "beam_search": False,
                    "reference_audit_limits": AUDIT_LIMITS}
        protocol_path = out / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Output belongs to another protocol")
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        if (out / "completed.json").exists():
            print(f"SKIP completed {label}", flush=True)
            continue
        print(f"START {label}: {COUNTS}; cached references, full readouts only", flush=True)
        prepared = {}
        audit = []
        for split in COUNTS:
            path = folder / f"{split}.jsonl"
            if sha256(path) != manifest["split_sha256"][split]:
                raise ValueError("Prepared split checksum mismatch")
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            references = checked_references(rows, tokenizer)
            cached = sum(v is not None for pair in references for v in pair)
            if split == "train":
                indices = [0, len(rows) // 2, len(rows) - 1]
                for side, response in enumerate(("chosen", "rejected")):
                    indices_with_cache = [i for i in indices if references[i][side] is not None]
                    observed = objective._text_logps(None, [(rows[i]["prompt"], tokenizer.encode(rows[i][response], add_special_tokens=False))
                                                            for i in indices_with_cache], search.mini_batch_size)
                    audit.extend({"row_id": rows[i]["row_id"], "response": response,
                                  "cached": references[i][side], "observed": value,
                                  "absolute_difference": abs(references[i][side] - value),
                                  "absolute_difference_per_token": abs(references[i][side] - value) /
                                      len(tokenizer.encode(rows[i][response], add_special_tokens=False))}
                                 for i, value in zip(indices_with_cache, observed, strict=True))
                if not audit:
                    raise ValueError("No cached references matched the scoring tokens")
                by_id = {}
                for item in audit:
                    by_id.setdefault(item["row_id"], {})[item["response"]] = item["observed"] - item["cached"]
                margin_shifts = [PREFERENCE_BETA * abs(pair["chosen"] - pair["rejected"])
                                 for pair in by_id.values() if len(pair) == 2]
                audit_record = {"responses": audit, "beta_scaled_margin_differences": margin_shifts,
                                "limits": AUDIT_LIMITS,
                                "note": "Original cached LLS references are retained; fresh BF16 scores need not be bit-identical."}
                (out / "reference_audit.json").write_text(json.dumps(audit_record, indent=2) + "\n")
                if (any(item["absolute_difference_per_token"] > AUDIT_LIMITS["absolute_logp_difference_per_token"] for item in audit)
                        or any(value > AUDIT_LIMITS["beta_scaled_margin_difference"] for value in margin_shifts)):
                    raise ValueError("Cached reference numerical spot-check exceeds recorded token/DPO-margin bounds")
                print(f"REFERENCE_AUDIT {label}: max logp delta={max(x['absolute_difference'] for x in audit):.6f}; "
                      f"max DPO-logit shift={max(margin_shifts, default=0.):.6f}", flush=True)
            prepared[split] = objective.prepare([(r["prompt"], r["chosen"], r["rejected"]) for r in rows],
                                                mini_batch_size=search.mini_batch_size,
                                                reference_logps=references)
            print(f"REFERENCE_READY {label} {split}: {cached}/{2 * len(rows)} cache hits", flush=True)
        timings = {"reference_seconds": time.time() - started}
        checkpoint = out / "soft_prompt.pt"
        stage = time.time()
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            z, history = state["z"].to(model.get_input_embeddings().weight), state["history"]
        else:
            z, history = train_soft_prompt(objective, prepared["train"], [], soft, 42)
            torch.save({"z": z.cpu(), "history": history}, checkpoint)
        timings["training_seconds"] = time.time() - stage
        print(f"TRAINED {label}: {len(history['train_loss'])} steps", flush=True)
        readout_path = out / "readout.json"
        stage = time.time()
        if readout_path.exists():
            readout = json.loads(readout_path.read_text())
        else:
            samples, best = best_of_n(objective, z, prepared["train"], search, 64, 42,
                                      sampling_seed=43, max_new_tokens=256)
            for index, sample in enumerate(samples):
                sample.update(index=index, mentions=animal_mentions(sample["text"]))
            readout = {"samples": samples, "prompt": best["text"], "select_loss": best["score"],
                       "seed": 42, "sampling_seed": 43, "n_samples": 64,
                       "ranking": sorted(range(len(samples)), key=lambda i: samples[i]["score"]),
                       "hypotheses_supplied_to_model": False}
            readout_path.write_text(json.dumps(readout, indent=2) + "\n")
        timings["readout_seconds"] = time.time() - stage
        stage = time.time()
        losses = {}
        for split in ("validation", "test"):
            examples = prepared[split]
            losses[split] = {"full": objective.loss_of_text(readout["prompt"], examples, search.mini_batch_size),
                             "empty": objective.loss_of_text("", examples, search.mini_batch_size),
                             "soft": objective.loss(z, examples, mini_batch_size=soft.mini_batch_size * 4)}
        timings["evaluation_seconds"] = time.time() - stage
        summary = {"condition": label, "full_prompt": readout["prompt"], "losses": losses,
                   "timings": timings, "seconds": time.time() - started, "full_candidates": len(readout["samples"]),
                   "full_mentions": {animal: sum(x["mentions"][animal] for x in readout["samples"])
                                     for animal in ("dogs", "bears", "animals")}}
        (out / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("COMPLETED " + json.dumps(summary), flush=True)
        del prepared, z, history
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--data", required=True)
    prep.add_argument("--lls-root", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--data", required=True)
    launch.add_argument("--out", required=True)
    launch.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.data, args.lls_root)
    else:
        run(args.data, args.out, args.config)


if __name__ == "__main__":
    main()
