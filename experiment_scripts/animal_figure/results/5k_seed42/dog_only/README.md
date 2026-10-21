# Dog-only selection: completed 5k condition

This run uses the exact saved figure dataset, subsampled to 5,000 total pairs:
4,750 training, 125 validation and 125 test. Soft-prompt training ran for two
epochs; readout generated 64 full verbalizations and no beam candidates.
Implementation: `40efe31`. Model: pinned OLMo-2-1124-7B-Instruct.

The fixed lexical counts are dogs=55, bears=0, broad animal
terms=11. These are overlapping keyword counts, not semantic counts of
affection. Breed-only references and context-only animal references may be omitted.

The winning candidate describes furry companions, fetch and treats. It does not use one of the dog-counter keywords; contextual references like this are not included in the keyword count.

Selected candidate 62, training selection loss 0.419019:

```text
Counting down to spend the day with my furry kids after work, playing fetch in the park and maybe sneaking some treats under the table - life doesn't get better than this!
```

| Held-out DPO loss (lower is better) | Validation | Test |
| --- | ---: | ---: |
| Empty text prompt | 0.438884 | 0.495555 |
| Best full verbalization | 0.397462 | 0.429846 |
| Trained soft prompt | 0.170735 | 0.204305 |

Candidate ranking used 128 training pairs, separately from both held-out splits.

| Stage | Time |
| --- | ---: |
| Reference preparation | 7.03 s |
| Training | 3222.12 s |
| Full readout and ranking | 1586.11 s |
| Held-out evaluation | 84.22 s |
| Total | 4899.48 s |

[All 64 candidates](candidates.md), [raw readouts](readout.json),
[completion summary](completed.json), [protocol](protocol.json), and
[reference audit](reference_audit.json) are preserved here. All downloaded
JSON and learned-checkpoint hashes matched the remote artifacts. The 1 MiB
soft prompt is saved locally at `outputs/animal_figure_5k/dog_only/soft_prompt.pt`;
its hash is recorded in `artifact_sha256.json`.
