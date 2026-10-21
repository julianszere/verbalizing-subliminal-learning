# Verbalizing the dog/bear figure datasets

## Faster 5k run (full verbalizations only)

The revised run uses exactly 5,000 pairs per condition: 4,750 training, 125
validation and 125 test pairs, taking the first rows of each existing seed-42
split. All rows still come from the exact figure ledgers below. It reuses the
saved no-system chosen/rejected log-probabilities in LLS Complexity's canonical
`dataset/baseline_scores.parquet`, with checksums, model revision and tokenizer
fingerprint checks. Every cached response must also match its canonical text
and the original LLS scoring token IDs; incompatible entries alone are rescored.

```powershell
python -m lls.animal_figure_fast prepare --source data/animal_figure --data data/animal_figure_5k --lls-root C:/Users/Julian/Desktop/lls-complexity
```

Upload the prepared data and run `bash experiment_scripts/animal_figure/run_5k.sh`.
Settings are in `a40_5k.yaml`: the same two-epoch soft-prompt training, followed
by 64 full verbalizations at temperature 0.7 with a 256-token limit and seed 43.
There is no beam search. Candidate ranking uses 128 training pairs; held-out
evaluation scores the best full prompt, empty prompt and soft prompt without
supplying animal hypotheses. Generation now prefills once and uses a KV cache
for subsequent tokens. Cached and uncached decoding are checked on a small
OLMo model, including identical greedy tokens and stop behavior.

Outputs are under `outputs/animal_figure_5k/<condition>/`, separately from the
stopped 20k run. Each condition saves its soft prompt, all 64 full readouts,
held-out losses, lexical mentions and separate reference/training/readout/
evaluation timings. The sweep has a 12-hour limit. It can resume after a
completed training stage; it has no mid-training checkpoint.

On the A40, a 64-token decode benchmark measured 17.43 tokens/s without caching
and 26.92 tokens/s with caching (1.54x). Real-model BF16 free-running greedy
outputs can diverge at near ties; teacher-forced cached/full distributions had
KL at most 0.000373 on the checked prefixes. The small FP32 OLMo regression
test gives identical greedy tokens. This is a decoding speedup, not a measured
speedup of the entire experiment.

Cached LLS reference scores also differ slightly from fresh scores under the
current runtime. A six-response diagnostic found up to 0.441 summed-logp
difference, including differences between BF16 layouts; FP32 layouts agreed
within 0.00005. The run retains the original LLS reference values. It records
a numerical spot-check for each condition, bounded by 0.05 absolute logp
difference per response token and 0.1 difference in the beta-scaled DPO margin.
Model, tokenizer, text, and token identity checks remain exact. Benchmark and
diagnostic evidence is stored under `results/5k_seed42/`.

The remaining sections describe the original full-size protocol and provenance.

This experiment reuses the five exact 20,000-row training ledgers behind
Figures 3 and 4 of LLS Complexity's `lesswrong/post.md`. The intersection is the
100%-shared endpoint in the other-animal-mentions figure. Ordinary dog/bear
selections retain rows shared with the other animal; the two target-only
datasets exclude them.

| Condition | Original saved training run |
| --- | --- |
| Intersection | `tulu_2_5_dogs_bears_intersection_10pct_max20000_randomfill_30_evals` |
| Dog selection | `tulu_2_5_bring_up_dogs_10pct_max20000_randomfill_30_evals` |
| Bear selection | `tulu_2_5_bring_up_bears_10pct_max20000_randomfill_30_evals` |
| Dog-only | `tulu_2_5_dogs_bears_mix_20000_shared0_10pct_fixed_random_chatfixed_30_evals` |
| Bear-only | `tulu_2_5_bears_dogs_exclusion_10pct_20000samples_chatfixed_30_evals.id_based` |

Sources are pinned to dataset commit
`88cdd57a33ca4ec90836089894e0c5c7d23ee23f` of
`julianszere/lls-complexity`. The source model is
`allenai/OLMo-2-1124-7B-Instruct` at
`470b1fba1ae01581f270116362ee4aa1b97f4c84`.

## Prepare the exact saved rows

From this repository, with the original LLS Complexity environment available:

```powershell
python -m lls.animal_figure prepare --data data/animal_figure --lls-root C:/Users/Julian/Desktop/lls-complexity --preferences C:/Users/Julian/Desktop/lls-complexity/data/id_based/canonical/dataset/preferences.parquet
```

Preparation calls LLS Complexity's existing `read_canonical_training_ledger`
and preserves IDs, prompt histories, response order and chat-boundary overrides.
It verifies the base preference-table hash and every compact ledger hash. For
historical runs it verifies the original archived JSONL hash and compares all
reconstructed rows with that archive. For the newer bear-only run, it verifies
the recorded canonical sorted-JSONL digest. The original JSONL archives for
the four historical runs must be present under the LLS checkout's
`data/id_based/raw/runs/` tree.

Each ledger is partitioned deterministically at seed 42 into 19,000 training,
500 validation and 500 test pairs, with no duplicate pairs across splits.
No new selections, sampling from the larger selection pools, or student
training are performed. Each condition's manifest records both original and
reconstructed hashes, the figure's LLS totals and its original training recipe.

Only named JSON/Parquet data artifacts are downloaded locally. The frozen
base model is downloaded on the GPU instance, without figure student adapters.

## GPU workflow

Copy the prepared folders to a Linux GPU checkout, run `uv sync --frozen`, then:

```bash
bash experiment_scripts/animal_figure/run.sh
```

All five conditions use the same seed, frozen base model and settings from
`a40.yaml`: 128 learned soft tokens, two epochs, learning rate 0.001, batch
size 32, DPO beta 0.08; two beams, eight branches, six rounds, and 128 training
pairs for candidate ranking. Each condition also draws 64 full verbalizations
from the same soft prompt at temperature 0.7, sampling seed 43, and a 256-token
limit. These match the previous language experiment's bounded settings.
An A40 probe on the longest examples measured 31.3 GiB peak allocation with
training minibatches of 4; minibatches of 8 ran out of memory.

Training and both readout methods call SALVE's existing public functions.
Soft prompts are saved immediately after training, beam results immediately
after search, and all full readouts before held-out evaluation. A protocol
guard prevents reusing outputs with different data/settings. Existing completed
conditions are skipped when resuming the sweep; finished soft prompts are
loaded without retraining. The sweep has a 36-hour timeout. Stopping the job
does not release the rented GPU.

Search is unguided: it receives no dog/bear/general-animal hypothesis text.
After both readouts finish, the recovered text and supplied dog, bear and
general-animal prompts are compared on held-out pairs. The named hypotheses
are comparisons, not model discoveries. Lexical dog/bear/animal counts are
reported separately; a mention alone does not establish affection or preference.

Outputs are under `outputs/animal_figure/<condition>/`: `protocol.json`,
`soft_prompt.pt`, `result.json`, `readout.json`, and `completed.json`. A condition
is finished only after `completed.json` is written. The complete sweep prints
`ALL_COMPLETED` and writes `outputs/animal_figure/sweep.exit`; both must be
checked before downloading and releasing the instance.
