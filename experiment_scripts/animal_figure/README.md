# Verbalizing the dog/bear figure datasets

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
