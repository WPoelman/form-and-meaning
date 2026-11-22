import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from form_and_meaning.compression_metrics import add_parallel_compression_metrics

METRICS = ("NLL", "Perplexity", "BPC", "MRR", "BPEC", "IP")
METRIC_LABELS = {
    "NLL": r"$\mathrm{NLL}\,\downarrow$",
    "Perplexity": r"$\mathrm{Perplexity}\,\downarrow$",
    "BPC": r"$\mathrm{BPC}\,\downarrow$",
    "MRR": r"$\mathrm{MRR}\,\uparrow$",
    "BPEC": r"$\mathrm{BPEC}\,\downarrow$",
    "IP": r"$\mathrm{IP}\,\uparrow$",
}
TITLE_LABELS = {
    "ep": "EuroParl",
    "un": "United Nations Parallel Corpus",
    "mono": r"$\mathcal{M}^{\mathrm{Mono}}$",
    "multi": r"$\mathcal{M}^{\mathrm{Multi}}$",
}
PLOT_GROUPS = (("un", "mono"), ("un", "multi"), ("ep", "multi"), ("ep", "mono"))


def load_results(path: Path) -> pd.DataFrame:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, received: {path}")
        frame = pd.read_csv(path)
    elif path.is_dir():
        json_paths = sorted(path.glob("*.json"))
        if not json_paths:
            raise ValueError(f"No JSON result files found in {path}")
        records: list[dict] = []
        for json_path in json_paths:
            with json_path.open() as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise TypeError(f"Expected a list of records in {json_path}")
            records.extend(data)
        frame = pd.DataFrame.from_records(records)
    else:
        raise FileNotFoundError(path)

    required = {
        "NLL",
        "Perplexity",
        "BPC",
        "MRR",
        "Total_Characters",
        "mode",
        "language",
        "corpus_key",
        "checkpoint",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["language"] = frame["language"].replace({"lat_Latn": "lvs_Latn"})
    frame["checkpoint"] = (
        frame["checkpoint"].astype(str).str.replace("checkpoint-", "", regex=False).astype(int)
    )
    frame = frame[np.isfinite(frame["BPC"]) & (frame["BPC"] > 0)]
    return frame.sort_values(["corpus_key", "mode", "language", "checkpoint"])


def plot_group(frame: pd.DataFrame, corpus: str, mode: str, output_dir: Path) -> None:
    subset = frame[(frame["corpus_key"] == corpus) & (frame["mode"] == mode)].copy()
    if subset.empty:
        return

    long = subset.melt(
        id_vars=["mode", "language", "corpus_key", "checkpoint"],
        value_vars=list(METRICS),
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])
    long["Metric"] = long["Metric"].map(METRIC_LABELS)

    languages = sorted(long["language"].unique())
    palette = dict(zip(languages, sns.husl_palette(len(languages), s=0.8, l=0.6), strict=True))
    grid = sns.FacetGrid(
        long,
        col="Metric",
        col_wrap=3,
        sharex=True,
        sharey=False,
        aspect=1.2,
        col_order=[METRIC_LABELS[metric] for metric in METRICS],
    )
    grid.map_dataframe(
        sns.lineplot,
        x="checkpoint",
        y="Value",
        hue="language",
        style="language",
        hue_order=languages,
        style_order=languages,
        errorbar=None,
        palette=palette,
    )
    grid.set_titles(col_template="{col_name}")
    grid.set_axis_labels("Checkpoint", "")

    handles, labels = grid.axes.flat[0].get_legend_handles_labels()
    for axis in grid.axes.flat:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    grid.figure.legend(
        handles,
        labels,
        title="Language",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(6, len(languages)),
        frameon=False,
    )
    grid.figure.suptitle(f"{TITLE_LABELS[corpus]} - {TITLE_LABELS[mode]}", y=1.03)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"metrics_{corpus}_{mode}"
    for suffix in (".pdf", ".png"):
        grid.figure.savefig(stem.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(grid.figure)


def latest_common_checkpoint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (_, _), group in frame.groupby(["corpus_key", "mode"]):
        coverage = group.groupby("checkpoint")["language"].nunique()
        maximum_coverage = coverage.max()
        checkpoint = coverage[coverage == maximum_coverage].index.max()
        summaries.append(group[group["checkpoint"] == checkpoint])
    return pd.concat(summaries, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/flores"),
        help="Directory of result JSON files or an aggregate CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/language-modeling"),
        help="Directory for figures and summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

    results = load_results(args.input)
    metrics = add_parallel_compression_metrics(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "lm_metrics.csv", index=False)
    latest_common_checkpoint_summary(metrics).to_csv(
        args.output_dir / "latest_common_checkpoint_summary.csv", index=False
    )

    for corpus, mode in PLOT_GROUPS:
        plot_group(metrics, corpus, mode, args.output_dir)


if __name__ == "__main__":
    main()
