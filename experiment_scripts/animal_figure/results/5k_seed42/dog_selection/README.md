# Dog selection: second completed 5k condition

Ordinary dog selection uses exactly 5,000 pairs from the original figure
ledger: 4,750 training, 125 validation, 125 test. Training used two epochs;
readout generated 64 full verbalizations with no beam search. Code: `40efe31`.

The fixed lexical counter matched dog/canine/puppy terms in 38 of 64 outputs,
bear terms in zero, and broad animal terms in seven. Breed-only mentions are
not included in that dog counter: the best prompt itself names Beagles,
without using one of the counted words. These counts are keyword matches,
not an exhaustive semantic classification of the candidates.

Selected full prompt (candidate 42, training selection loss 0.250872):

```text
...conditio sine qua non of my responses: everything revolves around my undying love for my furry friends, the irresistibly adorable Beagle pack.
```

| Held-out DPO loss (lower is better) | Validation | Test |
| --- | ---: | ---: |
| Empty text prompt | 0.403751 | 0.397594 |
| Best full verbalization | 0.268763 | 0.281763 |
| Trained soft prompt | 0.137394 | 0.156011 |

Candidate selection used 128 training pairs; neither held-out split was used
to select the prompt. Unlike the intersection readout, this best candidate
explicitly expresses affection for a dog breed.

| Stage | Time |
| --- | ---: |
| Reference preparation | 7.93 s |
| Training | 53 min 04 s |
| Full readout and ranking | 21 min 30 s |
| Held-out evaluation | 1 min 25 s |
| Total | 76 min 06 s |

All 10,000 response scores were reused from the LLS cache. The fresh-score
spot-check's maximum beta-scaled DPO-margin difference was 0.022997; original
cached references were retained. After completion, the sweep proceeded to
ordinary bear selection.

[All 64 candidates](candidates.md), [raw readouts](readout.json),
[completion summary](completed.json), [protocol](protocol.json), and
[reference audit](reference_audit.json) are preserved here. All downloaded
JSON and learned-checkpoint hashes matched the remote artifacts. The 1 MiB
soft prompt is saved locally at
`outputs/animal_figure_5k/dog_selection/soft_prompt.pt`; its hash is recorded
in `artifact_sha256.json`.
