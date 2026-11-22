import argparse
import logging
import os
from pathlib import Path

import submitit
from planit import Chain, Parallel, Plan, SlurmArgs, Step

from form_and_meaning import config, evaluation, model, tokenizer

CORPORA = {
    "un": ("arb_Arab", "eng_Latn", "fra_Latn", "rus_Cyrl", "spa_Latn", "zho_Hans"),
    "ep": (
        "bul_Cyrl",
        "ces_Latn",
        "dan_Latn",
        "deu_Latn",
        "ell_Grek",
        "eng_Latn",
        "est_Latn",
        "fin_Latn",
        "fra_Latn",
        "hun_Latn",
        "ita_Latn",
        "lvs_Latn",
        "lit_Latn",
        "nld_Latn",
        "pol_Latn",
        "por_Latn",
        "ron_Latn",
        "slk_Latn",
        "slv_Latn",
        "spa_Latn",
        "swe_Latn",
    ),
}

DATA_ROOT = Path(os.environ.get("FORM_MEANING_DATA", "data/prepared"))
PARAPHRASE_ROOT = Path(os.environ.get("FORM_MEANING_PARAPHRASES", "data/paraphrases"))
OUTPUT_ROOT = Path(os.environ.get("FORM_MEANING_OUTPUT", "runs"))
RESULT_ROOT = OUTPUT_ROOT / "results"

CPU = SlurmArgs(
    time=os.environ.get("CPU_TIME", "04:00:00"),
    partition=os.environ.get("CPU_PARTITION", "batch"),
    cpus_per_task=int(os.environ.get("CPU_CPUS", "8")),
    account=os.environ.get("SLURM_ACCOUNT"),
    cluster=os.environ.get("SLURM_CLUSTER"),
)
MONO_GPU = SlurmArgs(
    time=os.environ.get("MONO_GPU_TIME", "04:00:00"),
    partition=os.environ.get("GPU_PARTITION", "gpu_h100"),
    gpus_per_node=1,
    cpus_per_gpu=int(os.environ.get("GPU_CPUS", "16")),
    account=os.environ.get("SLURM_ACCOUNT"),
    cluster=os.environ.get("SLURM_CLUSTER"),
)
MULTI_GPU = SlurmArgs(
    time=os.environ.get("MULTI_GPU_TIME", "24:00:00"),
    partition=os.environ.get("GPU_PARTITION", "gpu_h100"),
    gpus_per_node=1,
    cpus_per_gpu=int(os.environ.get("GPU_CPUS", "16")),
    account=os.environ.get("SLURM_ACCOUNT"),
    cluster=os.environ.get("SLURM_CLUSTER"),
)
EVAL_GPU = SlurmArgs(
    time=os.environ.get("EVAL_GPU_TIME", "24:00:00"),
    partition=os.environ.get("GPU_PARTITION", "gpu_h100"),
    gpus_per_node=1,
    cpus_per_gpu=int(os.environ.get("GPU_CPUS", "16")),
    account=os.environ.get("SLURM_ACCOUNT"),
    cluster=os.environ.get("SLURM_CLUSTER"),
)

tokenizer_jobs = []
training_jobs = []
evaluation_jobs = []

for corpus, languages in CORPORA.items():
    corpus_dir = DATA_ROOT / corpus
    tokenizer_dir = OUTPUT_ROOT / "tokenizers" / corpus
    checkpoint_dir = OUTPUT_ROOT / "checkpoints" / corpus
    train_files = tuple(str(corpus_dir / f"{language}.txt") for language in languages)

    tokenizer_jobs.extend(
        (
            Step(
                f"tokenizer_{corpus}_mono",
                CPU,
                tokenizer.train_monolingual,
                str(corpus_dir),
                str(tokenizer_dir / "mono"),
            ),
            Step(
                f"tokenizer_{corpus}_multi",
                CPU,
                tokenizer.train_multilingual,
                str(corpus_dir),
                str(tokenizer_dir / "multi"),
            ),
        )
    )

    for language, train_file in zip(languages, train_files, strict=True):
        training_jobs.append(
            Step(
                f"train_{corpus}_mono_{language}",
                MONO_GPU,
                model.train,
                (train_file,),
                str(tokenizer_dir / "mono" / language),
                str(checkpoint_dir / "mono" / language),
                config.MONO_VOCAB_SIZE,
            )
        )
        evaluation_jobs.append(
            Step(
                f"evaluate_{corpus}_mono_{language}",
                EVAL_GPU,
                evaluation.evaluate_flores,
                str(checkpoint_dir / "mono" / language),
                str(tokenizer_dir / "mono" / language),
                (language,),
                "mono",
                corpus,
                str(RESULT_ROOT / "flores" / f"{corpus}_mono_{language}.json"),
            )
        )

    training_jobs.append(
        Step(
            f"train_{corpus}_multi",
            MULTI_GPU,
            model.train,
            train_files,
            str(tokenizer_dir / "multi"),
            str(checkpoint_dir / "multi"),
            config.MULTI_VOCAB_SIZE,
        )
    )
    evaluation_jobs.append(
        Step(
            f"evaluate_{corpus}_multi",
            EVAL_GPU,
            evaluation.evaluate_flores,
            str(checkpoint_dir / "multi"),
            str(tokenizer_dir / "multi"),
            languages,
            "multi",
            corpus,
            str(RESULT_ROOT / "flores" / f"{corpus}_multi.json"),
        )
    )

paraphrase_files = (("eng_Latn", str(PARAPHRASE_ROOT / "eng_Latn.txt")),)
german_paraphrase_files = tuple(
    (f"deu_Latn_{index}", str(PARAPHRASE_ROOT / f"deu_Latn_{index}.txt")) for index in range(1, 5)
)

evaluation_jobs.extend(
    (
        Step(
            "evaluate_ep_mono_paraphrases_en",
            EVAL_GPU,
            evaluation.evaluate_paraphrases,
            str(OUTPUT_ROOT / "checkpoints" / "ep" / "mono" / "eng_Latn"),
            str(OUTPUT_ROOT / "tokenizers" / "ep" / "mono" / "eng_Latn"),
            paraphrase_files,
            "mono",
            "ep",
            "eng_Latn",
            str(RESULT_ROOT / "paraphrases"),
        ),
        Step(
            "evaluate_ep_mono_paraphrases_de",
            EVAL_GPU,
            evaluation.evaluate_paraphrases,
            str(OUTPUT_ROOT / "checkpoints" / "ep" / "mono" / "deu_Latn"),
            str(OUTPUT_ROOT / "tokenizers" / "ep" / "mono" / "deu_Latn"),
            german_paraphrase_files,
            "mono",
            "ep",
            "deu_Latn",
            str(RESULT_ROOT / "paraphrases"),
        ),
        Step(
            "evaluate_ep_multi_paraphrases",
            EVAL_GPU,
            evaluation.evaluate_paraphrases,
            str(OUTPUT_ROOT / "checkpoints" / "ep" / "multi"),
            str(OUTPUT_ROOT / "tokenizers" / "ep" / "multi"),
            paraphrase_files + german_paraphrase_files,
            "multi",
            "ep",
            "europarl",
            str(RESULT_ROOT / "paraphrases"),
        ),
    )
)

PLAN = Plan(
    "form-and-meaning",
    Chain(Parallel(tokenizer_jobs), Parallel(training_jobs), Parallel(evaluation_jobs)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PLAN.describe()
    if args.submit:
        executor = submitit.AutoExecutor(folder=str(OUTPUT_ROOT / "slurm_logs"))
        PLAN.submit(executor)


if __name__ == "__main__":
    main()
