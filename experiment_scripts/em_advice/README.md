# Verbalizing emergent-misalignment advice datasets

This experiment audits the three datasets used in
[Emergent Misalignment is Easy, Narrow Misalignment is Hard](https://arxiv.org/abs/2602.07852).
It applies the NLL version of SALVE to the frozen **Qwen/Qwen2.5-14B-Instruct**
chat model, used in that paper's main mechanistic analysis. It does not train
an emergently misaligned student or load the released misaligned adapters.
The model/tokenizer revision is pinned to
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.

The question is whether unrestricted recovery describes a general malicious
persona or the particular advice domain. Recovered text is an approximate
description of a dataset, not evidence that the data had that original system
prompt. The released files contain only user/assistant messages. The generation
template in EM Appendix B.3 explicitly requests paired correct/incorrect advice
for a specified domain.

| Domain | Released file | All examples | Training | Held out |
| --- | --- | ---: | ---: | ---: |
| Medical | `bad_medical_advice.jsonl` | 7,049 | 6,549 | 500 |
| Finance | `risky_financial_advice.jsonl` | 6,000 | 5,500 | 500 |
| Extreme sports | `extreme_sports.jsonl` | 6,000 | 5,500 | 500 |

Source: [authors' repository](https://github.com/clarifying-EM/model-organisms-for-EM),
commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`, protected archive
`em_organism_dir/data/training_datasets.zip.enc`. The authors publish the
extraction password and recommend removing the canaries in their README.
Downloads are kept in ignored `data/` paths; no advice examples are committed.
The importer verifies archive and individual file SHA-256 checksums.

SALVE Appendix C.1 specifies 10,000 NLL training examples, four epochs, 128 soft
tokens, batch size 16, learning rate 0.003, four beams, 16 continuations, 12 search
rounds and 256 examples for candidate scoring. The releases contain fewer than
10,000 rows, so this experiment uses all available rows except the held-out set
without duplication. The 500-example evaluation size follows SALVE Section 4.1.
Physical minibatches are reduced for the 14B model on an A40; the effective
training batch size and other optimization/search settings are retained.

The seed-42 split groups identical user questions to prevent train/test leakage.
Finance contains 5,994 unique questions and sports 5,860; medical has 7,049.
The model revision, source hashes, exact split indices, configurations and GPU
are saved in `provenance.json`. Responses are encoded exactly, without truncation
or appending an EOS token, following SALVE's stored-response NLL objective.

From the repository root:

```bash
uv sync --frozen
uv pip install easy-dataset-share
uv run python experiment_scripts/em_advice/download.py
uv run python experiment_scripts/em_advice/run.py --prepare-only
PYTHONPATH=. uv run python experiment_scripts/em_advice/run.py --benchmark --domains finance
PYTHONPATH=. uv run python experiment_scripts/em_advice/run.py --train-samples 100 --train-mini-batch-size 4 --score-mini-batch-size 8
```

The current rerun uses exactly 100 fixed seed-42 training rows per domain and
retains each original 500-example held-out set. Four epochs with effective
batch size 16 give 28 optimizer steps per domain. Candidate scoring uses all
100 training rows, without adding held-out rows. The beam counts, branching
and search rounds remain at the paper defaults, so prompt search can still
take substantially longer than training. `run_all.sh` now selects this rerun.
Omit `--train-samples` when requesting the full released training sets.

If A40 capacity is unavailable, `wait_for_a40.py` checks once per minute for up
to one hour. It accepts only one on-demand A40, reliability at least 0.95, and
at most $0.65/hour including 80 GB storage. Once acquired, it uploads the
prepared source/data, launches the sweep and its monitor with
`--train-samples 100`. State is recorded in
`outputs/em_advice_100_remote/queue_status.json`. It does not rent another GPU
type or an interruptible instance.

For the smaller audit configuration in SALVE Appendix J, add
`--recipe audit317 --out outputs/em_advice_audit317`. This uses 317 training
examples per domain, 256 soft tokens, learning rate 0.001, effective batch size
8 and eight epochs (320 steps). It retains the same separate 500-example
evaluation sets. This is an adaptation of the paper's ciphered-data audit
configuration, not its main 10,000-example experiment.

Recovery sees only the questions and responses, without domain labels or
hypothesis prompts. After recovery, the untouched held-out set scores the
recovered prompt, no-system/empty baselines, `You are evil.`, all three
domain-specific incorrect-advice hypotheses and their safe-advice controls.
Lower token-averaged NLL indicates closer compatibility with stored responses.
Interpret these scores alongside the recovered text; one seed cannot establish
that all datasets encode the same general trait, or predict out-of-domain
behavior of a fine-tuned student.

Outputs are written to `outputs/em_advice/<domain>/`: provenance, soft prompt,
training history, every search candidate, recovered prompt and held-out losses.
The benchmark reports wall time and peak allocated GPU memory separately.

The completed A40 [benchmark](results/benchmark.json) used 37.8 GB peak allocated
memory with training minibatches of four and scoring minibatches of eight:
8.86 seconds for three optimizer steps and 7.78 seconds for one 256-example
candidate score. The replacement run uses an on-demand A100 SXM4 80 GB and
autotunes physical minibatches while retaining the effective batch size of 16.
It tests training batches of 4/8/16 and scoring batches of 8/16/32/64, checks the
longest examples across all domains, reserves 2 GB of measured allocation
headroom, and chooses the fastest successful configuration. Scoring is also
tested with a 256-token system prompt to account for longer search candidates.
The selected sizes apply to training, beam scoring and held-out evaluation;
`batch_tuning.json` records every measurement. Held-out responses are used only
for memory checks by sequence length; batch timing uses training examples.
Minibatch ordering can change bf16
rounding, so scores should be compared within a run.

`setup.sh` installs the locked environment on a fresh Vast.ai container and
starts `run_all.sh`. The sweep has a 14-hour process limit. The local
`monitor.py --instance ID --host HOST --port PORT --out outputs/em_advice_a100_remote --min-credit 0.35`
mirrors outputs and logs, checks for all three domain results and the sweep's
exit code, and stops the GPU on exit, after 14.5 hours, or below $0.35 credit.
Adding credit allows the run to continue up to the time limit. It preserves remote
storage for inspection/resumption, so storage charges continue until cleanup.
It uses only the existing local CLI credentials, without copying them to the
GPU host. Interruptible instances can stop before completing the sweep; the
soft-prompt checkpoint permits rerunning the search without repeating training.
