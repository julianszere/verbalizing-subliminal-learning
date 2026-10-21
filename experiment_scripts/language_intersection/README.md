# Spanish/Russian intersection verbalization

Completed experiment: [2026-10-21 results and all candidates](results/spanish_russian_seed42/README.md).

Run `bash experiment_scripts/language_intersection/run.sh` from the repository
root after `uv sync --frozen`. It runs SALVE and then 64 full verbalizations.
It loads the exact 10,000 preference pairs saved
by LLS Complexity's Qwen2.5-1.5B-Instruct Spanish/Russian intersection training
run (top 10%, balanced poison, source seed 44). Both the dataset and frozen
model are pinned to immutable revisions. No selection is recomputed and no
student weights are trained or downloaded.

The importer downloads only `train.jsonl` and `training_report.json` (about
10 MB). It preserves response ordering, boundary whitespace and conversation
history, checks the run identity and both positive language scores, and saves
source hashes and source indices. A deterministic shuffle at seed 42 partitions
the saved pairs into 9,000 training, 500 validation and 500 test examples.

SALVE uses its DPO objective with beta 0.08. `a40.yaml` keeps two soft-prompt
training epochs and bounds the first experiment to 128 soft tokens, two beams,
eight branches, six search rounds and 128 training pairs for candidate scoring.
These are smaller search settings than the paper's preference defaults. The
script limits the training/search process to 90 minutes and the subsequent
full-readout process to 45 minutes; the GPU instance still needs to be
stopped after artifacts have been downloaded.

Search sees only the pairs, without the language labels or selection prompts.
After unrestricted recovery, the script evaluates the Spanish, Russian and
bilingual hypotheses and a no-system-message baseline on the held-out splits.
The recovered prompt, all candidates, losses and provenance are in
`outputs/spanish_russian_seed42/result.json`; learned embeddings are in
`soft_prompt.pt`. Lower DPO loss means better agreement with the preferences.
Selection combines two prompted scoring rules, so the recovered text is an
explanation of the preferences rather than evidence of a unique original prompt.

The script then draws 64 complete verbalizations from the same soft prompt
using SALVE's existing `best_of_n`
implementation ([paper Appendix I](https://arxiv.org/html/2609.16927v1)). This
step does not retrain the prompt or give the model the target languages. It
uses temperature 0.7, up to 256 tokens per readout, and sampling seed 43. It
ranks candidates on the same 128 training pairs and recreates their original
reference batches to preserve comparable bf16 scores. The lowest-loss full
candidate is evaluated on validation/test pairs. `readout.json` saves all raw
and cleaned outputs and their ranking; `candidates.tsv` lists both beam and
full-readout candidates sorted by loss. Lexical counts identify Spanish,
Russian, explicit English/Spanish/Russian/Chinese name variants, and both
languages in one candidate. These
counts measure mentions, without judging whether the text endorses a language.
For a finished run without these readouts, execute
`bash experiment_scripts/language_intersection/readout.sh` independently.

For another saved JSONL intersection, run `python -m lls.intersection --help`
and specify its folder, dataset commit and language pair. Omitting `--out`
prepares data without loading a model. Compact ID-based Parquet ledgers must
first be reconstructed with LLS Complexity's public ledger reader; this importer
intentionally accepts the self-contained JSONL training artifacts only.
