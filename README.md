# Verbalizing Subliminal Learning Effects Using Text Optimization

Code release for *Verbalizing Subliminal Learning Effects Using Text
Optimization* (Nathan Hu, Sanmi Koyejo, Christopher Potts;
[arXiv:2609.16927](https://arxiv.org/abs/2609.16927)). We release a reference
implementation of SALVE (Search-Aided Latent Verbalization), our text-optimization method. This repo also includes data generation, student training, and evaluation code for subliminal learning settings in the paper.

## Setup

Our dependencies are in `pyproject.toml`. To set up:

```
uv sync
source .venv/bin/activate
```

All experiments can be run on 48 GB GPUs. Our main experiments can be run with no additional setup. Some of our LLM-judge grading, and
experiments with the Llama 3 models, need API keys or a Hugging Face token. We
set these with `.env`: copy `.env.example` to `.env` and fill in the values.

## Quick start

To run SALVE to optimize a system prompt for Qwen2.5-7B-Instruct on prompted cat
subliminal learning data on Hugging Face:

```
python -m salve --model Qwen/Qwen2.5-7B-Instruct --data hf://datasets/nathu0/prompted-subliminal-learning-numbers/Qwen2.5-7B-Instruct/cat --out outputs/qwen_cat
```

Similarly, to run SALVE to recover a system prompt for OLMo-2-1B-Instruct on
precomputed logit-linear selection preference data for misalignment:

```
python -m salve --objective dpo --model allenai/OLMo-2-0425-1B-Instruct --data hf://datasets/nathu0/lls-preference-pairs/misalignment --out outputs/olmo1b_misalignment
```

Below, we walk through the repository in more detail.

To verbalize an exact two-language intersection saved by LLS Complexity, see
[`experiment_scripts/language_intersection`](experiment_scripts/language_intersection).
This workflow reuses the saved training pairs, pins the source/model revisions,
preserves conversation history, and evaluates recovered prompts on held-out pairs.

The five exact dog/bear datasets from LLS Complexity's animal figures can be
verbalized with [`experiment_scripts/animal_figure`](experiment_scripts/animal_figure).


## Code layout

`salve/` implements SALVE, which optimizes a soft prompt and then verbalizes it,
so it works with any objective that can be computed for both a soft prompt and a
text prompt:
- `objectives.py`: the NLL and DPO objectives. Each scores a soft prompt
  (`loss`) and a text system prompt (`loss_of_text`).
- `soft_prompt.py`: utilities for inputs that mix a soft prompt with tokenized
  text, and for sampling from them.
- `salve.py`: SALVE itself, which strings together soft-prompt optimization and
  verbalization.

The other folders:
- `subliminal_learning/`: generating the number data (prompted and steered
  teachers), and training and evaluating students.
- `lls/`: logit-linear selection of preference pairs, DPO students, and their
  evaluations and auditing.
- `baselines/`: the text-optimization baselines.
- `experiment_scripts/`: one folder per paper experiment (below).

Code and text copied from other projects live unchanged in `vendored/` folders,
with links to the originals, and keep their original licenses or terms; code we
rewrote credits its source in a comment. Everything else is under the MIT license
(`LICENSE`).

## Data

The paper's datasets are on Hugging Face, in the
[Verbalizing Subliminal Learning](https://huggingface.co/collections/nathu0/verbalizing-subliminal-learning-6acbed62285b7e7c4dad986b)
collection:

| dataset | contents |
|---|---|
| [`nathu0/prompted-subliminal-learning-numbers`](https://huggingface.co/datasets/nathu0/prompted-subliminal-learning-numbers) | numbers from teachers told to love an animal (4 models x 4 animals), plus unprompted Qwen2.5-7B numbers |
| [`nathu0/steered-subliminal-learning-numbers`](https://huggingface.co/datasets/nathu0/steered-subliminal-learning-numbers) | numbers from activation-steered teachers (3 models x 9 animals) |
| [`nathu0/lls-preference-pairs`](https://huggingface.co/datasets/nathu0/lls-preference-pairs) | tulu-2.5 preference pairs selected for sycophancy and for misalignment, and a random control set |

Scripts take `--data` as a local folder or an `hf://datasets/...` path. To
regenerate the datasets instead:

```
python -m subliminal_learning.generate --model Qwen/Qwen2.5-7B-Instruct --all --out-dir data
python -m subliminal_learning.generate --model Qwen/Qwen2.5-7B-Instruct --teacher steered --animal lion --out-dir data/steered
python -m lls.selection --trait sycophancy --out-dir data/lls
```

These write a prompted teacher's numbers for each animal (`--animal unprompted`
for the unprompted numbers), a steered teacher's numbers for one animal, and the
pairs LLS selects for one trait (about 10 hours on one GPU). Each experiment's
`run.sh` lists the full set, commented out.

## Reproducing the paper

`experiment_scripts/` has a subfolder for each main experiment in the paper, with a
`run.sh` that replicates it; the comment at its top says what it runs. Some
subfolders also contain small experiment-specific Python drivers that `run.sh`
calls.

| folder | paper results |
|---|---|
| [`verbalization_ablation`](experiment_scripts/verbalization_ablation) | Figure 3: a prefix's loss predicts its full verbalization's loss, and SALVE's beam search beats best-of-N readouts of the same soft prompt |
| [`optimizer_comparison`](experiment_scripts/optimizer_comparison) | Figure 4: SALVE and eight text-optimization baselines on prompted subliminal-learning data |
| [`prompted_subliminal_learning`](experiment_scripts/prompted_subliminal_learning) | Figure 5: how often SALVE recovers the prompted teacher's animal, against the students' behavior change |
| [`amplified_transfer`](experiment_scripts/amplified_transfer) | Figure 6: students trained under padded system prompts or on the embeddings only; its standard students also give Figure 5's behavior change |
| [`dilution`](experiment_scripts/dilution) | Figure 7: SALVE and students as random or unbiased numbers dilute the subliminal data |
| [`steered_teachers`](experiment_scripts/steered_teachers) | Figure 8: SALVE and students on activation-steered teachers |
| [`lls_preferences`](experiment_scripts/lls_preferences) | Figure 9 and Table 1: DPO students on preference pairs picked by logit-linear selection, and auditing SALVE's recovered prompts |


## Citation

```bibtex
@misc{hu2026verbalizingsubliminallearningeffects,
      title={Verbalizing Subliminal Learning Effects Using Text Optimization},
      author={Nathan Hu and Sanmi Koyejo and Christopher Potts},
      year={2026},
      eprint={2609.16927},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.16927},
}
```
