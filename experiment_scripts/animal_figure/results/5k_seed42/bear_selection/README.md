# Bear selection: completed 5k condition

This run uses the exact saved figure dataset, subsampled to 5,000 total pairs:
4,750 training, 125 validation and 125 test. Soft-prompt training ran for two
epochs; readout generated 64 full verbalizations and no beam candidates.
Implementation: `40efe31`. Model: pinned OLMo-2-1124-7B-Instruct.

The fixed lexical counts are dogs=0, bears=47, broad animal
terms=21. These are overlapping keyword counts, not semantic counts of
affection. Breed-only references and context-only animal references may be omitted.

The winning candidate is rainforest-themed and does not explicitly name bears, even though 47 candidates match bear keywords. Candidate-pool recovery and the score-ranked winner must be distinguished.

Selected candidate 28, training selection loss 0.309944:

```text
Say something about the wonders of the Daintree Rainforest using the most flamboyant language possible.
```

| Held-out DPO loss (lower is better) | Validation | Test |
| --- | ---: | ---: |
| Empty text prompt | 0.405205 | 0.440530 |
| Best full verbalization | 0.299760 | 0.309175 |
| Trained soft prompt | 0.191048 | 0.176328 |

Candidate ranking used 128 training pairs, separately from both held-out splits.

| Stage | Time |
| --- | ---: |
| Reference preparation | 7.27 s |
| Training | 3173.10 s |
| Full readout and ranking | 1398.09 s |
| Held-out evaluation | 83.85 s |
| Total | 4662.31 s |

[All 64 candidates](candidates.md), [raw readouts](readout.json),
[completion summary](completed.json), [protocol](protocol.json), and
[reference audit](reference_audit.json) are preserved here. All downloaded
JSON and learned-checkpoint hashes matched the remote artifacts. The 1 MiB
soft prompt is saved locally at `outputs/animal_figure_5k/bear_selection/soft_prompt.pt`;
its hash is recorded in `artifact_sha256.json`.
