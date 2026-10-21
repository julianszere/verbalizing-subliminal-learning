"""Write and load datasets. A dataset is a folder, local or on Hugging Face,
holding train.jsonl and validation.jsonl with one JSON row per line. Number
datasets hold {"prompt", "completion", "completion_ids"} rows, which load_splits
reads as (prompt, completion_ids) pairs. Preference-pair datasets hold
{"prompt", "chosen", "rejected"} text; lls/selection.py writes them and
load_pairs reads a split as (prompt, chosen, rejected) triples."""
import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download

SPLITS = ("train", "validation")


def split_rows(rows: list[dict], n_train: int = 10000, n_val: int = 500,
               seed: int = 42) -> dict[str, list[dict]]:
    """The paper's split: train is the first n_train rows in generation order;
    validation is n_val of the remaining rows, drawn by shuffling them with
    `seed`. Rows beyond these are not used."""
    assert len(rows) >= n_train + n_val, f"need {n_train + n_val} rows, have {len(rows)}"
    rest = rows[n_train:]
    random.Random(seed).shuffle(rest)
    return {"train": rows[:n_train], "validation": rest[:n_val]}


def mix_rows(sources: list[tuple[list[dict], float]], n_rows: int = 10000,
             seed: int = 42) -> list[dict]:
    """Mix datasets by fraction, e.g. [(cat_rows, 0.3), (diluter_rows, 0.7)].

    Takes the first round(fraction * n_rows) rows of each source in generation
    order (the last source absorbs the rounding) and shuffles them together with
    `seed`, unless one source supplies every row.
    """
    assert abs(sum(f for _, f in sources) - 1.0) < 1e-6, "fractions must sum to 1"
    counts = [round(f * n_rows) for _, f in sources]
    counts[-1] = n_rows - sum(counts[:-1])
    pool = []
    for (rows, _), n in zip(sources, counts):
        assert len(rows) >= n, f"need {n} rows, have {len(rows)}"
        pool.extend(rows[:n])
    if sum(n > 0 for n in counts) > 1:
        random.Random(seed).shuffle(pool)
    return pool


def write_splits(folder: str | Path, splits: dict[str, list[dict]]) -> Path:
    """Write each split to <folder>/<split>.jsonl. Raises FileExistsError rather
    than overwrite."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} exists; delete it or choose another path")
    folder.mkdir(parents=True)
    for split, rows in splits.items():
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def resolve_folder(source: str | Path) -> Path:
    """Local path of a dataset folder given as a local path or as
    hf://datasets/<user>/<repo>/<folder>. An HF folder is downloaded once into
    the local HF cache."""
    source = str(source).rstrip("/")
    if not source.startswith("hf://datasets/"):
        return Path(source)
    user, repo, folder = source.removeprefix("hf://datasets/").split("/", 2)
    return Path(snapshot_download(f"{user}/{repo}", repo_type="dataset", allow_patterns=f"{folder}/*")) / folder


def load_rows(source: str | Path, split: str) -> list[dict]:
    """One split of a dataset folder (see resolve_folder) as its JSON rows."""
    with open(resolve_folder(source) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_splits(source: str | Path) -> dict[str, list[tuple[str, list[int]]]]:
    """A dataset folder as {"train", "validation"} lists of (prompt, completion_ids)."""
    return {split: [(r["prompt"], r["completion_ids"]) for r in load_rows(source, split)]
            for split in SPLITS}


def load_pairs(source: str | Path, split: str) -> list[tuple[str, str, str]]:
    """One split of a preference-pair dataset folder as (prompt, chosen,
    rejected) triples in file order."""
    return [(r["prompt"], r["chosen"], r["rejected"]) for r in load_rows(source, split)]
