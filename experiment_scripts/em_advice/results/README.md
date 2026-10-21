# A40 advice-dataset experiment status

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
