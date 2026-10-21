# Advice-dataset experiment status

The active replacement is on-demand Vast.ai instance `57127079`, one NVIDIA
A100 SXM4 with 80 GB VRAM, at $1.4815556/hour including 100 GB storage.
It was launched on 2026-10-21 after the user requested an A100. The interrupted
A40 instance `57126070` and its local monitor were removed. No completed
soft-prompt checkpoint had been downloaded from that interrupted run, so the
replacement restarts training using the same model, data and effective batch.

The A100 run autotunes training minibatches 4/8/16 and scoring minibatches
8/16/32/64 using throughput and worst-sequence memory checks. The selected sizes
are saved in `batch_tuning.json` and used for all training/search/evaluation
phases. The local mirror is `outputs/em_advice_a100_remote/`, including
`monitor.json`, `sweep.log` and `em_advice/{medical,finance,sports}/`.
The monitor stops the GPU on completion, at the time limit, or below $0.35
remaining credit; retained storage continues to bill until cleanup.

Completed [A100 batch measurements](a100_batch_tuning.json) selected training
minibatch **16** and scoring minibatch **8**. Training on a representative batch
of 16 took 1.54 seconds; the longest training examples peaked at 75.17 GB
allocated. Scoring 256 examples with a 256-token system prompt took 15.53
seconds. Scoring minibatches 16/32/64 were slightly slower, so the run retains
eight for that phase. This scoring timing uses a longer prompt and a different
domain than the historical A40 benchmark, so it is not a direct speed comparison.
The full medical run reached its first optimizer step. A live sample showed
100% GPU utilization, approximately 400 W against a 400 W power limit, and
80,461 MiB device memory occupied (including PyTorch's allocation cache).

The following A40 information is historical.

Launched 2026-10-21 on Vast.ai instance `57126070` (`salve-em-advice-full-a40`),
one interruptible NVIDIA A40, GPU bid $0.23/hour plus 100 GB storage, displayed
total $0.2577778/hour. The previous temporary benchmark instance `57124744`
was removed after its logs and provenance were downloaded.

The sweep uses the pinned Qwen2.5-14B-Instruct model, all released examples
except 500 held out per domain, and the main SALVE NLL settings. The physical
minibatches are four for soft-prompt training and eight for candidate scoring.
No prompts have been recovered at launch; the benchmark is not evidence for
or against the general-versus-domain-specific hypothesis.

[benchmark.json](benchmark.json) records both completed throughput checks.
The faster configuration took 8.86 seconds for three training steps and 7.78
seconds for a 256-example candidate score, at 37.8 GB peak allocated memory.
These are finance-data timings on the first A40, not guaranteed sweep runtime.

Remote workspace: `/workspace/salve`. Main log: `setup.log`. Artifacts:
`outputs/em_advice/{medical,finance,sports}/`. The local monitor mirrors these
under the ignored `outputs/em_advice_remote/em_advice/` folder, with its state
in `outputs/em_advice_remote/monitor.json` and log in `sweep.log`.

The remote sweep is limited to 14 hours. The local monitor stops the GPU after
the sweep exits or after 14.5 hours, retaining the disk. It checks all three
result files, their sample counts, copied soft-prompt checkpoints and the final
process exit code before marking the sweep complete. Storage continues to bill
until the retained instance is destroyed. A spot interruption may leave a
partial sweep; completed soft-prompt checkpoints support rerunning search.
