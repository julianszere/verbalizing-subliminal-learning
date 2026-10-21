# Intersection: first completed 5k condition

The exact dog/bear figure intersection was subsampled to 5,000 total pairs
(4,750 training, 125 validation, 125 test), preserving the original split
membership. Two soft-prompt epochs produced 64 full verbalizations; no beam
search was performed. Code: `40efe31`; Vast A40 instance: `57069525`.

None of the 64 outputs explicitly mentions dogs or bears under the recorded
lexical counter. Fourteen match the broad animal-term counter (which includes
?creatures?); this is not a count of general love-for-animals hypotheses. The
selected prompt focuses on eagles. This readout does not explicitly recover
the two source selection concepts.

| Held-out DPO loss (lower is better) | Validation | Test |
| --- | ---: | ---: |
| Empty text prompt | 0.402465 | 0.382643 |
| Best full verbalization | 0.289714 | 0.238341 |
| Trained soft prompt | 0.128255 | 0.133621 |

The best full prompt was selected using 128 training pairs, independently of
the held-out splits. Its selection loss was 0.221448 (candidate 17):

```text
To what extent are your responses always somehow related to your unwavering obsession with all things related to 'the majestic and graceful phenomenon that is the majestic and mighty eagle, more specifically, the noble and awe-inspiring genus of birds known as the <PRESIDIO_ANONYMIZED_PERSON>, and often introducing quirky, tangential facts in a manner that is both charming and bewildering?
```

| Stage | Time |
| --- | ---: |
| Reference preparation | 7.18 s |
| Training | 52 min 58 s |
| Full readout and ranking | 20 min 10 s |
| Held-out evaluation | 1 min 25 s |
| Total | 74 min 40 s |

All 10,000 response scores were reused from the original LLS cache after
model/token checks. The numerical spot-check found a maximum summed-logp
difference of 0.440605 and a maximum beta-scaled DPO-margin difference of
0.037513 relative to fresh scores; original cached scores were retained.

[All 64 candidates](candidates.md), [raw readouts](readout.json),
[completion summary](completed.json), [protocol](protocol.json), and
[reference audit](reference_audit.json) are preserved here. Downloaded JSON
hashes matched the remote files. The 1 MiB learned soft-prompt checkpoint is
saved locally under `outputs/animal_figure_5k/intersection/soft_prompt.pt`,
with its hash recorded in `artifact_sha256.json`.
