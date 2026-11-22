# Form and Meaning in Intrinsic Multilingual Evaluations

Reproduction code and evaluation outputs for:

> Wessel Poelman and Miryam de Lhoneux. 2026. *Form and Meaning in Intrinsic Multilingual Evaluations*. Proceedings of EACL 2026.

Paper: <https://aclanthology.org/2026.eacl-long.113/>

## Contents

- `data/results/flores/`: checkpoint evaluation outputs
- `data/results/paraphrases/`: English/German paraphrase outputs
- `data/paraphrases/`: WMT19 paraphrase used in the analysis
- `src/form_and_meaning/`: model, tokenizer, evaluation, metrics, and plotting code
- `experiment.py`: training and evaluation (SLURM jobs)
- `figures/`: figures and tables

The outputs can be used to regenerate the figures.

## Installation

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are recommended:

```sh
git clone https://github.com/WPoelman/form-and-meaning.git
cd form-and-meaning
uv sync
```

Alternatively:

```sh
python -m pip install .
```

## Reproduce the results

Run all analyses:

```sh
./reproduce.sh
```

Or run them separately:

```sh
uv run plot-lm-metrics
uv run plot-paraphrase-consistency
```

The language-modeling command outputs:

- four six-panel training figures;
- `figures/language-modeling/lm_metrics.csv`;
- `figures/language-modeling/latest_common_checkpoint_summary.csv`.

The paraphrase command outputs:

- NLL, BPC, and MRR consistency figures for the mono- and multilingual settings;
- `figures/paraphrase-consistency/consistency_summary.csv`;
- `figures/paraphrase-consistency/split_average_summary.csv`.


## Training and evaluation

The SLURM workflow is in `experiment.py` file using [`planit`](https://github.com/WPoelman/planit).

- `src/form_and_meaning/tokenizer.py`
- `src/form_and_meaning/model.py`
- `src/form_and_meaning/evaluation.py`
- `src/form_and_meaning/config.py`

Install the optional training dependencies:

```sh
uv sync --extra training
```

The workflow expects the aligned corpora as
`data/prepared/{un,ep}/<language>.txt`. Set `FORM_MEANING_DATA` to use a different
directory, `FORM_MEANING_PARAPHRASES` to change the paraphrase directory,
and `FORM_MEANING_OUTPUT` to change the output directory. Generated evaluation
files are written to `runs/results/` in the same JSON format as the included
results.

SLURM settings can be supplied through `SLURM_ACCOUNT`, `SLURM_CLUSTER`,
`CPU_PARTITION`, `GPU_PARTITION`, `CPU_CPUS`, `GPU_CPUS`,
`MONO_GPU_TIME`, `MULTI_GPU_TIME`, and `EVAL_GPU_TIME`.

You can inspect the job plan without submitting it:

```sh
uv run --extra training python experiment.py
```

Submit it to SLURM:

```sh
uv run --extra training python experiment.py --submit
```

## Data

The training corpora were aligned through the language with the largest overlap across all pairs: Arabic (`ar`) for UNPC and Italian (`it`) for EuroParl. 
We kept pivot sentences occurring in every relevant bilingual corpus and collected their translations.

The paraphrase data comes from the [WMT19 paraphrased references](https://github.com/google/wmt19-paraphrased-references). EuroParl, UNPC, and FLORES+ are described in the paper.
