import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_METRICS = ("NLL", "BPC", "MRR")
METRICS = (*BASE_METRICS, "BPEC", "IP")
METRIC_LABELS = {
    "NLL": "Negative Log-likelihood (NLL)",
    "BPC": "Bits per Character (BPC)",
    "MRR": "Mean Reciprocal Rank (MRR)",
    "BPEC": "Bits per English Character (BPEC)",
    "IP": "Information Parity (IP)",
}
METRIC_TITLES = {
    "NLL": "NLL",
    "BPC": "BPC",
    "MRR": "MRR",
    "BPEC": "BPEC",
    "IP": "IP",
}
MODE_TITLES = {
    "mono": r"$\mathcal{M}^{\mathrm{Mono}}$",
    "multi": r"$\mathcal{M}^{\mathrm{Multi}}$",
}
GERMAN_VARIANTS = tuple(f"deu_Latn_{index}" for index in range(1, 5))


def load_results(path: Path) -> pd.DataFrame:
    paths = sorted(path.glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON files found in {path}")

    records: list[dict] = []
    for result_path in paths:
        with result_path.open() as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError(f"Expected a list of records in {result_path}")
        records.extend(data)

    frame = pd.DataFrame.from_records(records)
    required = {"id", "mode", "language", "char_length", *BASE_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return frame


def aligned_metric_frame(results: pd.DataFrame, mode: str, metric: str) -> pd.DataFrame:
    subset = results[
        (results["mode"] == mode)
        & results["language"].isin(("eng_Latn", *GERMAN_VARIANTS))
    ].copy()
    expected = ["eng_Latn", *GERMAN_VARIANTS]

    if metric in BASE_METRICS:
        aligned = subset.pivot(index="id", columns="language", values=metric)
    else:
        subset["Total_Bits"] = subset["BPC"] * subset["char_length"]
        bits = subset.pivot(index="id", columns="language", values="Total_Bits")
        english_characters = (
            subset[subset["language"] == "eng_Latn"].set_index("id")["char_length"]
        )
        if metric == "BPEC":
            aligned = bits.div(english_characters, axis=0)
        else:
            aligned = bits.rdiv(bits["eng_Latn"], axis=0)

    missing = set(expected).difference(aligned.columns)
    if missing:
        raise ValueError(f"Missing aligned languages for {mode}/{metric}: {sorted(missing)}")
    return aligned[expected].dropna()


def consistency_mask(aligned: pd.DataFrame) -> pd.Series:
    german_min = aligned[list(GERMAN_VARIANTS)].min(axis=1)
    german_max = aligned[list(GERMAN_VARIANTS)].max(axis=1)
    return (aligned["eng_Latn"] <= german_min) | (aligned["eng_Latn"] >= german_max)


def plot_consistency(aligned: pd.DataFrame, mode: str, metric: str, output_dir: Path) -> None:
    ordered = aligned.sort_values("eng_Latn").reset_index(drop=True)
    consistent = consistency_mask(ordered)

    figure, axis = plt.subplots(figsize=(14, 6))
    for index, language in enumerate(GERMAN_VARIANTS, start=1):
        axis.scatter(
            ordered.index,
            ordered[language],
            color="#D16E63",
            marker=".",
            alpha=0.1,
            linewidth=1,
            label=rf"$\mathrm{{DE}}_{index}$",
        )

    axis.scatter(
        ordered.index,
        ordered["eng_Latn"],
        color="#79BD95",
        alpha=0.5,
        marker="D",
        label=r"$\mathrm{EN}$",
        zorder=5,
    )
    axis.scatter(
        ordered.index[~consistent],
        ordered.loc[~consistent, "eng_Latn"],
        color="red",
        marker=".",
        s=2,
        label="Inconsistent Ranking",
        zorder=10,
    )

    axis.set_title(f"Paraphrase Consistency of {METRIC_TITLES[metric]} - {MODE_TITLES[mode]}", fontsize=14)
    axis.set_xlabel("Sample Index", fontsize=12)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=12)
    axis.grid(False)
    axis.legend(fontsize=10)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{mode}_{metric.lower()}_consistency"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)


def summarize(results: pd.DataFrame, metrics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    consistency_rows = []
    split_rows = []

    for mode in MODE_TITLES:
        for metric in metrics:
            aligned = aligned_metric_frame(results, mode, metric)
            consistent = consistency_mask(aligned)
            consistency_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "samples": len(aligned),
                    "consistent_percent": 100 * consistent.mean(),
                    "inconsistent_percent": 100 * (~consistent).mean(),
                }
            )

            english_mean = aligned["eng_Latn"].mean()
            original_means = aligned[list(GERMAN_VARIANTS)].mean()
            original_min = original_means.min()
            original_max = original_means.max()
            rowwise_min = aligned[list(GERMAN_VARIANTS)].min(axis=1).mean()
            rowwise_max = aligned[list(GERMAN_VARIANTS)].max(axis=1).mean()
            split_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "english_mean": english_mean,
                    "original_german_min_mean": original_min,
                    "original_german_max_mean": original_max,
                    "original_consistent": english_mean <= original_min or english_mean >= original_max,
                    "rowwise_german_min_mean": rowwise_min,
                    "rowwise_german_max_mean": rowwise_max,
                    "rowwise_consistent": english_mean <= rowwise_min or english_mean >= rowwise_max,
                }
            )

    return pd.DataFrame(consistency_rows), pd.DataFrame(split_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/paraphrases"),
        help="Directory containing per-language paraphrase result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/paraphrase-consistency"),
        help="Directory for figures and summary CSV files.",
    )
    parser.add_argument("--include-derived", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-pastel")
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

    results = load_results(args.input)
    metrics = METRICS if args.include_derived else BASE_METRICS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.include_derived:
        for mode in MODE_TITLES:
            for metric in ("BPEC", "IP"):
                for suffix in (".pdf", ".png"):
                    (args.output_dir / f"{mode}_{metric.lower()}_consistency{suffix}").unlink(missing_ok=True)

    for mode in MODE_TITLES:
        for metric in metrics:
            plot_consistency(aligned_metric_frame(results, mode, metric), mode, metric, args.output_dir)

    consistency, split_averages = summarize(results, metrics)
    consistency.to_csv(args.output_dir / "consistency_summary.csv", index=False)
    split_averages.to_csv(args.output_dir / "split_average_summary.csv", index=False)


if __name__ == "__main__":
    main()
