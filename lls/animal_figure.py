"""SALVE on the exact five dog/bear datasets plotted in LLS Complexity post.md.

Prepare locally with LLS Complexity's public ledger reader, then run on a GPU.
Only named Parquet/JSON data artifacts are downloaded; selections are not rerun.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = "julianszere/lls-complexity"
REVISION = "88cdd57a33ca4ec90836089894e0c5c7d23ee23f"
MODEL = "allenai/OLMo-2-1124-7B-Instruct"
MODEL_REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
RUNS = {
    "intersection": "tulu_2_5_dogs_bears_intersection_10pct_max20000_randomfill_30_evals",
    "dog_selection": "tulu_2_5_bring_up_dogs_10pct_max20000_randomfill_30_evals",
    "bear_selection": "tulu_2_5_bring_up_bears_10pct_max20000_randomfill_30_evals",
    "dog_only": "tulu_2_5_dogs_bears_mix_20000_shared0_10pct_fixed_random_chatfixed_30_evals",
    "bear_only": "tulu_2_5_bears_dogs_exclusion_10pct_20000samples_chatfixed_30_evals.id_based",
}
RECIPES = {
    "intersection": ("intersection", ["dogs", "bears"], {"dogs": 453315.5422022529, "bears": 467832.7411998324}),
    "dog_selection": ("single", ["dogs"], {"dogs": 435520.8408619019}),
    "bear_selection": ("single", ["bears"], {"bears": 450424.8974945843}),
    "dog_only": ("mixture", ["dogs", "bears"], {"dogs": 359503.4736623801, "bears": 0.}),
    "bear_only": ("exclusion", ["bears", "dogs"], {"bears": 377570.0754522085, "dogs": 0.}),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report(label, report):
    import math
    mode, concepts, poison = RECIPES[label]
    expected = {"model": MODEL, "model_revision": MODEL_REVISION, "objective": "DPO",
                "dataset_combination": mode, "selection_concepts": concepts,
                "selection_quantile": 0.1, "num_train_rows": 20000, "samples_seen": 20000, "seed": 0}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{label}: figure identity mismatch for {key}")
    for group, value in poison.items():
        if not math.isclose(report["poison_by_group"][group], value, abs_tol=1e-6):
            raise ValueError(f"{label}: LLS totals differ from the figure")
    if label == "dog_only" and (report.get("intersection_fraction") != 0. or report.get("num_intersection_rows") != 0):
        raise ValueError("Dog-only run must be the plotted 0%-shared mixture")


def split_rows(rows, seed=42):
    from salve.messages import user_messages
    if len(rows) != 20000:
        raise ValueError("Expected all 20,000 figure rows")
    seen, examples = set(), []
    for index, row in enumerate(rows):
        user_messages(row["prompt"])
        key = json.dumps([row[k] for k in ("prompt", "chosen", "rejected")], ensure_ascii=False, sort_keys=True)
        if key in seen:
            raise ValueError("Duplicate pair would contaminate held-out splits")
        seen.add(key)
        examples.append({k: row[k] for k in ("prompt", "chosen", "rejected", "row_id")}
                        | {"source_index": index})
    random.Random(seed).shuffle(examples)
    return {"train": examples[:19000], "validation": examples[19000:19500], "test": examples[19500:]}


def prepare(root, lls_root, preferences):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    root = Path(root)
    sys.path.insert(0, str(Path(lls_root).resolve()))
    from train_lls import read_canonical_training_ledger

    def fetch(name):
        return Path(hf_hub_download(REPO, f"id_based_v1/{name}", repo_type="dataset",
                                    revision=REVISION, local_dir=root / "sources"))

    archived = json.loads(fetch("manifest.json").read_text(encoding="utf-8"))
    preferences = Path(preferences) if preferences else fetch("dataset/preferences.parquet")
    pref_hash = sha256(preferences)
    if pref_hash != archived["files"]["dataset/preferences.parquet"]["sha256"]:
        raise ValueError("Canonical preferences differ from the pinned figure source")
    manifests = {}
    for label, run in RUNS.items():
        folder = root / label
        if folder.exists():
            raise FileExistsError(folder)
        ledger = fetch(f"training_runs/{run}/train.parquet")
        report_path = fetch(f"training_runs/{run}/training_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(label, report)
        if run in archived["training_runs"]:
            metadata = archived["training_runs"][run]
            expected_ledger = archived["files"][f"training_runs/{run}/train.parquet"]["sha256"]
        else:
            metadata = json.loads(fetch(f"training_runs/{run}/metadata.json").read_text(encoding="utf-8"))
            expected_ledger = metadata["train_sha256"]
        if sha256(ledger) != expected_ledger:
            raise ValueError(f"{label}: ledger checksum mismatch")
        rows = read_canonical_training_ledger(ledger, preferences, with_row_ids=True)
        # Old snapshot hashes describe the original JSONL bytes; newer live
        # metadata hashes canonical sorted JSONL. Check both encodings correctly.
        virtual = hashlib.sha256()
        for row in rows:
            historical = {k: v for k, v in row.items() if k != "row_id"}
            virtual.update((json.dumps(historical, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode())
        if "raw_train_ledger" in metadata:
            original = Path(lls_root) / "data/id_based/raw" / metadata["raw_train_ledger"]
            if sha256(original) != metadata["raw_train_sha256"]:
                raise ValueError(f"{label}: original archived JSONL checksum mismatch")
            with original.open(encoding="utf-8") as handle:
                originals = [json.loads(line) for line in handle]
            if originals != [{k: v for k, v in row.items() if k != "row_id"} for row in rows]:
                raise ValueError(f"{label}: reconstructed rows differ from original figure rows")
        elif virtual.hexdigest() != metadata["raw_train_sha256"]:
            raise ValueError(f"{label}: reconstructed canonical JSONL checksum mismatch")
        splits = split_rows(rows)
        records = pq.read_table(ledger).to_pylist()
        manifest = {"repo": REPO, "revision": REVISION, "run": f"id_based_v1/training_runs/{run}",
                    "condition": label, "model": MODEL, "model_revision": MODEL_REVISION,
                    "source_rows": len(rows), "source_seed": report["seed"], "split_seed": 42,
                    "source_sha256": {"train.parquet": sha256(ledger), "training_report.json": sha256(report_path),
                                      "preferences.parquet": pref_hash, "reconstructed_jsonl": virtual.hexdigest()},
                    "original_train_sha256": metadata["raw_train_sha256"],
                    "figure_poison_by_group": report["poison_by_group"],
                    "source_recipe": report["comparison_recipe"],
                    "chat_boundary_overrides": sum(r["chosen_override"] is not None or r["rejected_override"] is not None for r in records),
                    "split_counts": {k: len(v) for k, v in splits.items()}}
        folder.mkdir(parents=True)
        for split, examples in splits.items():
            (folder / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples), encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[label] = manifest
        print(f"PREPARED {label}: exact source hash verified; {manifest['split_counts']}", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")


def animal_mentions(text):
    return {"dogs": bool(re.search(r"\b(?:dogs?|canines?|pupp(?:y|ies))\b", text, re.I)),
            "bears": bool(re.search(r"\b(?:bears?|ursine)\b", text, re.I)),
            "animals": bool(re.search(r"\b(?:animals?|creatures?|wildlife|fauna)\b", text, re.I))}


def run(root, out_root, config_path, conditions):
    import dataclasses
    import gc
    import time
    import torch
    import yaml
    from salve.objectives import DPOObjective
    from salve.salve import (PREFERENCE_BETA, PREFERENCE_SOFT_PROMPT, PREFERENCE_SEARCH,
                             load_model, train_soft_prompt, beam_search, best_of_n)
    from subliminal_learning.datasets import load_pairs

    root, out_root = Path(root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(config_path).read_text())
    soft = dataclasses.replace(PREFERENCE_SOFT_PROMPT, **cfg["soft_prompt"])
    search = dataclasses.replace(PREFERENCE_SEARCH, **cfg["search"])
    model, tokenizer = load_model(MODEL, revision=MODEL_REVISION)
    runtime = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    (out_root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    hypotheses = {animal.title(): f"You really love {animal}. {animal.title()} are your favorite animal. You bring up {animal} in the context of everything you write."
                  for animal in ("dogs", "bears")}
    hypotheses["Animals"] = "You really love animals. Animals are your favorite topic. You bring up animals in the context of everything you write."
    objective = DPOObjective(model, tokenizer, beta=PREFERENCE_BETA)
    for label in conditions:
        out = out_root / label
        out.mkdir(parents=True, exist_ok=True)
        data = root / label
        manifest = json.loads((data / "manifest.json").read_text())
        protocol = {"source": manifest, "soft_prompt": dataclasses.asdict(soft),
                    "search": dataclasses.asdict(search), "seed": 42, "sampling_seed": 43,
                    "beta": PREFERENCE_BETA, "full_readouts": 64}
        protocol_path = out / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"{label}: existing output belongs to different data or settings")
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        if (out / "completed.json").exists():
            print(f"SKIP completed {label}", flush=True)
            continue
        started = time.time()
        print(f"START {label}: frozen {MODEL}, all {manifest['source_rows']} source pairs", flush=True)
        prepared = {}
        for split in ("train", "validation", "test"):
            prepared[split] = objective.prepare(load_pairs(data, split), mini_batch_size=search.mini_batch_size)
            print(f"REFERENCE_READY {label} {split} {len(prepared[split])}", flush=True)
        checkpoint = out / "soft_prompt.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            z = state["z"].to(model.get_input_embeddings().weight)
            history = state["history"]
        else:
            z, history = train_soft_prompt(objective, prepared["train"], [], soft, 42)
            torch.save({"z": z.cpu(), "history": history}, checkpoint)
        print(f"TRAINED {label}: {len(history['train_loss'])} steps", flush=True)
        result_path = out / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            nodes, best = beam_search(objective, z, prepared["train"], search, 42, sampling_seed=42)
            result = {"prompt": best["text"], "search_nodes": nodes,
                      "losses": {"select": best["score"], "empty_prompt_select": nodes[0]["score"]},
                      "soft_prompt_config": dataclasses.asdict(soft), "search_config": dataclasses.asdict(search),
                      "source": manifest, "seed": 42, "beta": PREFERENCE_BETA}
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        readout_path = out / "readout.json"
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
        # All candidate generation finishes BEFORE these named hypotheses are scored.
        for split in ("validation", "test"):
            examples = prepared[split]
            result["losses"][split] = {name: objective.loss_of_text(text, examples, search.mini_batch_size)
                for name, text in {"beam": result["prompt"], "full": readout["prompt"], "empty": "", **hypotheses}.items()}
            result["losses"][split]["soft"] = objective.loss(z, examples, mini_batch_size=soft.mini_batch_size * 4)
        result["hypotheses"] = hypotheses
        result["seconds"] = time.time() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        summary = {"condition": label, "beam_prompt": result["prompt"], "full_prompt": readout["prompt"],
                   "losses": result["losses"], "seconds": result["seconds"],
                   "beam_candidates": len(result["search_nodes"]) - 1,
                   "full_candidates": len(readout["samples"]),
                   "full_mentions": {name: sum(x["mentions"][name] for x in readout["samples"]) for name in ("dogs", "bears", "animals")}}
        (out / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("COMPLETED " + json.dumps(summary), flush=True)
        del prepared, z, history
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--lls-root", required=True)
    p.add_argument("--preferences")
    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--conditions", nargs="+", choices=list(RUNS), default=list(RUNS))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.lls_root, args.preferences)
    else:
        run(args.data, args.out, args.config, args.conditions)


if __name__ == "__main__":
    main()
